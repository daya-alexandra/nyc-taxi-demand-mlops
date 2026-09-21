Commands
========

The root ``Makefile`` is the canonical command index.

* ``make install`` installs the locked development environment.
* ``make repro`` reproduces the DVC graph.
* ``make repro-real`` requires the real raw dataset.
* ``make validate`` runs formatting, lint, security checks and tests.
* ``make api`` starts FastAPI and the Web UI.
* ``make mlflow`` starts the local MLflow server.
* ``make docker-build`` builds a runnable image.
* ``make compose-up`` starts API, MLflow, Prometheus and Grafana.
* ``make k8s-apply`` applies the local Minikube overlay.
* ``make k8s-apply-base`` applies the GHCR/Argo CD base manifests.
