# Подготовка к защите NYC Taxi Demand MLOps

Этот файл — не рекламный текст, а шпаргалка: что именно делает проект, в каком порядке это показать и где честно назвать ограничения.

## Суть проекта в 30 секунд

«Я прогнозирую число посадок Yellow Taxi в заданной NYC taxi zone на следующий час. DVC воспроизводит весь путь от raw/sample data до features, модели, batch predictions и drift report. MLflow хранит параметры, метрики, артефакты и версии модели. FastAPI обслуживает online inference, Prometheus и Grafana дают observability, Docker/Kubernetes упаковывают и запускают систему, а GitHub Actions и Argo CD автоматизируют проверки, публикацию image и deployment».

## Как устроен один prediction

1. Клиент или форма UI отправляет 15 признаков в `POST /predict`.
2. Pydantic проверяет типы и диапазоны полей.
3. API проверяет, что joblib model существует и десериализуется.
4. Имена API-полей переводятся в те же feature names, которые использовались в training.
5. Модель возвращает demand; отрицательное значение ограничивается нулём.
6. Простые бизнес-правила добавляют `demand_spike`, `weather_risk` или `missing_lag_signal`.
7. Prediction попадает в in-memory history, а counters/histogram обновляются для Prometheus.

## Код по слоям

| Файл | Что важно объяснить |
| --- | --- |
| `src/data/make_dataset.py` | Два явных источника: real TLC + Open-Meteo или deterministic sample; provenance записывается отдельно |
| `src/features/build_features.py` | Calendar features и lags считаются отдельно внутри каждой зоны; `shift(1)` защищает от подглядывания в текущий target |
| `src/models/train_model.py` | Time split, HGBR с Poisson loss, model vs naive metrics, MLflow run и Registry |
| `src/models/predict_model.py` | Batch prediction и absolute error — вход для concept drift |
| `src/monitoring/calculate_drift.py` | Равные соседние окна, PSI и recent MAE ratio |
| `src/api/app.py` | HTTP contract, readiness, inference, metrics и настоящий background retrain |
| `dvc.yaml` / `dvc.lock` | Декларация DAG против зафиксированного состояния конкретного запуска |
| `.github/workflows/ci.yml` | Проверяет код, pipeline, image, HTTP и manifests до merge |
| `.github/workflows/cd.yml` | После зелёного CI публикует immutable SHA image и при наличии secrets вызывает Argo CD |
| `Dockerfile` | Multistage build, model bootstrap, минимальный runtime, non-root user |
| `infra/` | Compose для локального stack; Kustomize base/overlay; Argo Application |

## Почему выбран такой ML baseline

`HistGradientBoostingRegressor`:

- хорошо работает с нелинейностями и взаимодействиями calendar/weather/lag features;
- быстрее классического gradient boosting на десятках тысяч строк;
- не требует scaling;
- поддерживает Poisson loss, логичный для неотрицательного count target;
- остаётся понятным baseline, а не чрезмерно сложной нейросетью.

Главное доказательство полезности — не красивый R² сам по себе, а сравнение на одном и том же future test-window с наивными прогнозами. На sample model MAE `4.338`, лучший naive lag-168h MAE `5.913`; улучшение около 27%.

## Почему нет leakage

- Split идёт по уникальным timestamps, а не случайным строкам. Все зоны одного часа остаются в одной части.
- Train — только ранние часы, test — только более поздние.
- Lag features используют `shift`, поэтому target текущего часа не попадает в его же признаки.
- Препроцессинг погоды не использует target.

Замечание для более строгого production-варианта: rolling/lag values для online inference должен готовить feature store или отдельный поток актуальных агрегатов; сейчас их передаёт клиент demo API.

## DVC и MLflow — не одно и то же

| Инструмент | За что отвечает здесь |
| --- | --- |
| Git | Код, небольшие JSON/HTML reports, manifests и история review |
| DVC | DAG, hashes больших dataset/model artifacts, selective recomputation и remote storage |
| MLflow Tracking | История training runs: params, metrics, artifacts, dataset source |
| MLflow Registry | Именованные версии модели, которые можно переводить между lifecycle stages/aliases |

Фраза для защиты: «DVC отвечает на вопрос, из каких версий данных и кода получился artifact; MLflow — как прошёл эксперимент и какая версия модели зарегистрирована».

## Три вида drift

### Data drift

Изменилось распределение входных признаков `P(X)`. Здесь для каждого feature считается Population Stability Index:

\[
PSI = \sum_i (p_i - q_i) \ln\frac{p_i}{q_i}
\]

Reference bins строятся по quantiles, затем сравниваются доли reference/current. Пороги: warning `0.1`, critical `0.2`. `month` и `day_of_month` исключены: они гарантированно изменяются с календарём и давали бы бессмысленный alarm.

### Target drift

Изменилось распределение фактического спроса `P(y)`. Используется PSI `trip_count` в тех же соседних окнах. Этот расчёт возможен только после появления actual target.

### Concept drift

Изменилось соответствие между features и target, то есть `P(y|X)`. Proxy в проекте — отношение MAE двух соседних recent windows. Warning начинается с `1.25`, critical — с `1.5`.

В текущем sample data drift critical из-за сезонного изменения температуры, а target/concept drift остаются ok. Правильная интерпретация: alert требует анализа, но сам по себе не доказывает, что модель деградировала.

## Health endpoints

- `/health/live`: процесс HTTP жив; Kubernetes не должен restart pod только из-за отсутствующей модели.
- `/health/ready`: pod готов принимать inference; проверяется не только наличие файла, но и его десериализация/структура.
- `/health`: подробная диагностика для человека и UI.

Это лучше одного всегда-зелёного `/health`: corrupted или missing model не может дать ложную готовность.

## Что делает кнопка retrain

Кнопка вызывает `POST /api/retrain`. API:

1. неблокирующим lock запрещает два одновременных запуска;
2. сохраняет `queued` status и audit event;
3. в background task запускает фиксированную команду `python -m dvc repro --force` без shell;
4. пишет stdout/stderr в `reports/retrain.log`;
5. проверяет новый model artifact, очищает API caches и сохраняет `succeeded` или `failed`.

Честное ограничение: для production это должен быть отдельный authenticated worker/job с очередью, object storage, rollout новой версии и rollback. In-process background task здесь нужен для учебной end-to-end демонстрации.

## CI против CD

CI не только запускает четыре старых unit tests. Он устанавливает lock, проверяет style/security/18+ tests, полностью воспроизводит DVC DAG, строит Docker image, запускает container и делает реальные HTTP-запросы, затем валидирует Compose и rendered Kubernetes manifests.

CD запускается только после успешного CI на `main`. Image получает тег commit SHA, поэтому deployment можно воспроизвести и откатить; `latest` оставлен для удобства. Argo CD sync выполняется только при наличии cluster credentials.

## GitOps и Argo CD

Git хранит желаемое состояние Kubernetes. Argo CD сравнивает его с фактическим cluster state, применяет изменения, self-heal исправляет ручной drift, а prune удаляет ресурсы, убранные из Git. CD передаёт Argo immutable SHA image, после чего sync/wait проверяет health.

Важно: `infra/argocd/application.yaml` не доказывает, что внешний cluster сейчас работает. Для фактического deploy нужны установленный Argo CD, доступный cluster, credentials и доступ к GHCR image.

## Сценарий live demo на 7–10 минут

Перед защитой выполнить preflight из следующего раздела. На самой защите удобно держать три терминала.

### 1. Репозиторий и pipeline — 2 минуты

```bash
python -m dvc dag
python -m dvc status
python -m dvc metrics show
```

Показать `dvc.yaml`, назвать пять stages. Если есть время на 30–60 секунд:

```bash
NYC_TAXI_DATA_MODE=sample python -m dvc repro --force
```

В PowerShell: `$env:NYC_TAXI_DATA_MODE="sample"; python -m dvc repro --force`.

### 2. MLflow — 1–2 минуты

Терминал 1:

```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Открыть <http://127.0.0.1:5000>, показать experiment, model/naive metrics, artifacts и зарегистрированную версию.

### 3. API и UI — 2 минуты

Терминал 2:

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Открыть `/docs`, выполнить `/health/ready` и `/predict`, затем `/ui`: prediction, experiments, monitoring и HTML report. Нажимать retrain стоит только если есть ещё 30–60 секунд на ожидание.

### 4. Observability/deployment — 2 минуты

Если Docker работает, вместо отдельных процессов заранее поднять:

```bash
docker compose -f infra/docker-compose.yml up --build
```

Показать Prometheus Targets (`api` должен быть UP) и готовый Grafana dashboard. Затем открыть CI YAML, Dockerfile, Kustomize local overlay и Argo Application; необязательно ждать живой Minikube rollout во время рассказа.

## Preflight за день до защиты

```bash
python --version
python -m pip install -r requirements-dev-lock.txt
make validate
NYC_TAXI_DATA_MODE=sample python -m dvc repro --force
python -m dvc status
node --check src/web/app.js
```

Затем проверить:

- branch/PR отправлены в GitHub и CI зелёный;
- GHCR package доступен cluster или настроен `imagePullSecret`;
- `/health/ready`, `/predict`, `/metrics`, `/api/drift`, `/reports/drift` возвращают ожидаемые status codes;
- MLflow UI видит последний run и Registry version;
- если обещается Docker demo — Compose поднимается именно на ноутбуке защиты;
- если обещается Minikube — заранее выполнить `minikube image load` и `kubectl apply -k infra/k8s/overlays/local`;
- реальные TLC metrics показываются только после отдельного real run и `data_profile.is_synthetic == false`.

## Типовые вопросы преподавателя

### Почему sample вообще допустим?

Он не заменяет исследование на реальных данных. Он решает инженерную задачу: CI и чистый clone обязаны пройти весь DAG без частного `C:\dvc-storage`. Provenance явно маркирует источник. Финальные научные выводы нужно получить в real mode.

### Почему не random split?

Задача временная. Random split перемешал бы будущее с прошлым и дал оптимистичную оценку. Split по часам моделирует реальное прогнозирование будущего.

### Почему DVC output не в Git?

Parquet и joblib бинарные и крупные. Git хранит код и `.dvc` metadata/lock, а DVC content-addressed cache/remote — сами версии artifacts.

### Что означает registry version?

Это не версия исходного кода. Это новая зарегистрированная model version, связанная с конкретным MLflow run, его metrics и artifacts.

### Что произойдёт при изменении feature code?

DVC увидит новый hash dependency, пересчитает `build_features`, затем downstream train, predict и drift. Незатронутые upstream stages можно взять из cache.

### Почему HTTP 503, а не 500, если нет модели?

Сервис существует, но временно не способен обслуживать inference. 503 правильно сообщает orchestrator/load balancer, что instance не ready.

### Что именно мониторит Prometheus?

Доступность модели, количество/latency запросов, сумму predicted demand, anomaly flags, retrain requests, offline MAE/RMSE/R² и data/target/concept drift values.

### Можно ли автоматически retrain при любом PSI alert?

Нет. PSI может отражать сезонность или корректное изменение бизнеса. Сначала проверяются качество, target/concept drift, data quality и причина изменения; затем принимается решение о retrain и rollout.

### Как сделать rollback?

CD публикует image с immutable commit SHA. Можно вернуть предыдущий SHA в Kustomize/Argo image override и синхронизировать; model Registry отдельно хранит версии моделей.

### Это production-ready?

Это production-like учебный контур. Для production не хватает auth/TLS, внешнего feature store, durable prediction store, managed MLflow DB/artifact storage, alert routing, отдельного retraining orchestrator, canary rollout и SLA/load testing.

## Чего не говорить

- Не называть synthetic sample «реальными поездками NYC».
- Не говорить, что Argo deploy выполнен, если был только создан manifest или job был skipped.
- Не утверждать, что `/api/retrain` — полноценный production orchestration.
- Не делать вывод о падении качества только по data drift.
- Не путать prediction anomaly flags с statistical drift.

Самая сильная позиция на защите — показать работающий end-to-end контур и спокойно назвать его границы.
