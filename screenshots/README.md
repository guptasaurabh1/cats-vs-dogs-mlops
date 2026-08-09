# Screenshots to capture on your own laptop

These cannot be produced in an automated review environment - capture them
yourself while following Section 14 ("Screen-Recording Script") of
reports/MLOps_Assignment2_Final_Report.docx, and drop them in this folder
before zipping your final submission.

| Filename (suggested)                          | What to capture                                                         |
|------------------------------------------------|---------------------------------------------------------------------------|
| github_repo_home.png                          | Your GitHub repository home page, showing the file tree                  |
| ci_workflow_list.png                          | GitHub Actions tab, list of workflow runs                                |
| ci_pipeline_running.png                       | A CI run in progress (test/lint/build jobs visible)                      |
| ci_pipeline_run_details.png                   | A completed CI run, expanded, green checks for test/lint/build/push      |
| docker_build.png                              | Terminal: `docker build ...` completing successfully                      |
| docker_run.png                                | Terminal: `docker run ...` plus a curl call to /health                    |
| docker_desktop_kubernetes_settings.png        | Docker Desktop with Kubernetes enabled (or `kind get clusters` output)    |
| kubectl_context_check.png                     | Terminal: `kubectl config current-context`                                |
| kubectl_pods.png / kubs_pods_running.png      | Terminal: `kubectl get pods` showing 2 Running replicas                   |
| k8s_service.png                               | Terminal: `kubectl get svc cats-vs-dogs-classifier`                       |
| kub_smoke_test.png                            | Terminal: `deploy/smoke_test.py` output showing ALL SMOKE TESTS PASSED    |
| swagger_ui.png                                | Browser: http://localhost:8000/docs (FastAPI Swagger UI)                  |
| predict_response.png                          | Browser or curl: a real /predict response body on a real cat/dog photo   |
| mlflow_runs.png                               | MLflow UI: the experiment's run list                                     |
| mlflow_run_detail.png                         | MLflow UI: one run's detail page (params, metrics, artifacts tab)        |
| prometheus_or_metrics.png                     | Browser or curl: GET /metrics output                                     |

If you don't set up Grafana/Prometheus as a separate stack, the /metrics
endpoint screenshot alone satisfies M5's monitoring requirement.
