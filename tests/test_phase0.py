def test_environment_initialization():
    """Verify SmartCret package can be imported and version is set."""
    import smartcret
    assert smartcret.__version__ == "0.1.0"
