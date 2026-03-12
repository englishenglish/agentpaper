import json
import sys
import types
import datetime as dt
import typing


def _install_module(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


# Stub optional third-party/runtime modules so unit tests can run offline.
dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: False
_install_module("dotenv", dotenv)

yaml = types.ModuleType("yaml")
yaml.YAMLError = Exception
yaml.safe_load = lambda *_args, **_kwargs: {}
yaml.dump = lambda obj, **_kwargs: json.dumps(obj, ensure_ascii=False, indent=2)
_install_module("yaml", yaml)

chromadb = types.ModuleType("chromadb")
chromadb_api = types.ModuleType("chromadb.api")
chromadb_types = types.ModuleType("chromadb.api.types")
chromadb_types.Embedding = list
chromadb_types.PyEmbedding = list
chromadb_types.OneOrMany = list
_install_module("chromadb", chromadb)
_install_module("chromadb.api", chromadb_api)
_install_module("chromadb.api.types", chromadb_types)

if not hasattr(typing, "ParamSpecArgs"):
    typing.ParamSpecArgs = typing.Any

if "zoneinfo" not in sys.modules:
    zoneinfo = types.ModuleType("zoneinfo")

    class ZoneInfo(dt.tzinfo):
        def __init__(self, name: str):
            self._name = name
            self._offset = dt.timedelta(hours=8) if name == "Asia/Shanghai" else dt.timedelta(0)

        def utcoffset(self, _value):
            return self._offset

        def dst(self, _value):
            return dt.timedelta(0)

        def tzname(self, _value):
            return self._name

    zoneinfo.ZoneInfo = ZoneInfo
    _install_module("zoneinfo", zoneinfo)
