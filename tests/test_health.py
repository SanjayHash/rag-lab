from raglab.health import get_project_status

def test_project_status():
    status = get_project_status()

    assert status["application"] == "RAG Lab"
    assert status["status"] == "running"
    assert status["version"] == "0.1.0"