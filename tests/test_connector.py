from litserve.connector import _Connector


def test_connector():
    connector = _Connector()
    assert isinstance(connector.accelerator, str)
