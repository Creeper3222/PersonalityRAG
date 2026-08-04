from __future__ import annotations

import logging
from pathlib import Path

from personalityrag.logger import (
    LOGGER_NAME,
    WebLogBuffer,
    configure_logging,
    flush_logging,
    get_log_buffer,
    logger,
    sanitize_log_message,
)


def test_web_log_buffer_limits_entries_bytes_and_entry_size():
    buffer = WebLogBuffer(max_entries=3, max_bytes=2000, max_entry_bytes=30)
    record = logging.LogRecord(
        LOGGER_NAME,
        logging.INFO,
        __file__,
        1,
        "x" * 500,
        (),
        None,
    )
    buffer.append_record(record)
    item = buffer.get_entries()[0]
    assert item["level"] == "INFO"
    assert "truncated" in item["message"]
    assert len(item["message"]) < 520

    for index in range(10):
        buffer.append_record(
            logging.LogRecord(
                LOGGER_NAME,
                logging.INFO,
                __file__,
                index,
                "line-%s",
                (index,),
                None,
            )
        )
    assert len(buffer.get_entries()) <= 3
    assert buffer.summary()["bytes"] <= buffer.summary()["max_bytes"]


def test_configure_logging_writes_rotated_file_and_web_buffer(tmp_path):
    log_file = tmp_path / "logs" / "personalityrag.log"
    configure_logging(
        log_file,
        level_name="INFO",
        file_max_bytes=180,
        file_backup_count=1,
        web_max_entries=20,
        web_max_bytes=4096,
        web_max_entry_bytes=1024,
    )
    for index in range(20):
        logger.info("rotation-check-%s", index)
    flush_logging()

    assert log_file.exists()
    assert (tmp_path / "logs" / "personalityrag.log.1").exists()
    assert get_log_buffer().summary()["entry_count"] <= 20


def test_web_buffer_honors_configured_level(tmp_path):
    log_file = tmp_path / "logs" / "personalityrag.log"
    configure_logging(
        log_file,
        level_name="INFO",
        web_max_entries=20,
        web_max_bytes=4096,
        web_max_entry_bytes=1024,
    )

    logger.debug("web-only-debug-record")
    logger.info("persistent-info-record")
    flush_logging()

    entries = get_log_buffer().get_entries()
    assert not any(
        item["level"] == "DEBUG" and item["message"] == "web-only-debug-record"
        for item in entries
    )
    file_text = log_file.read_text(encoding="utf-8")
    assert "web-only-debug-record" not in file_text
    assert "persistent-info-record" in file_text


def test_log_sanitizer_redacts_secrets():
    text = sanitize_log_message(
        "api_key=secret authorization=BearerToken "
        "prag_abcdefghijklmnopqrstuvwxyz "
        "psk-abcdefghijklmnopqrstuvwxyz012345 "
        "pkb-abcdefghijklmnopqrstuvwxyz012345 "
        "data:image/png;base64,aGVsbG8= "
        "https://example.test/api/v1/knowledge-libraries/text_media_v1/"
        "demo/assets/image/content?exp=123&sig=secret"
    )
    assert "secret" not in text
    assert "BearerToken" not in text
    assert "prag_abcdefghijklmnopqrstuvwxyz" not in text
    assert "psk-abcdefghijklmnopqrstuvwxyz012345" not in text
    assert "pkb-abcdefghijklmnopqrstuvwxyz012345" not in text
    assert "aGVsbG8=" not in text
    assert "exp=123" not in text
    assert "[redacted]" in text


def test_configured_logging_redacts_all_persistent_and_web_sinks(tmp_path):
    log_file = tmp_path / "logs" / "personalityrag.log"
    configure_logging(
        log_file,
        level_name="INFO",
        web_max_entries=20,
        web_max_bytes=4096,
        web_max_entry_bytes=1024,
    )

    logger.info(
        "adapter failure pkb-abcdefghijklmnopqrstuvwxyz012345 "
        "data:image/png;base64,aGVsbG8="
    )
    flush_logging()

    file_text = log_file.read_text(encoding="utf-8")
    web_text = "\n".join(
        item["message"] for item in get_log_buffer().get_entries()
    )
    for output in (file_text, web_text):
        assert "pkb-abcdefghijklmnopqrstuvwxyz012345" not in output
        assert "aGVsbG8=" not in output
        assert "[redacted]" in output


def test_launcher_never_prints_the_api_key_value():
    source = (Path(__file__).resolve().parents[1] / "run.py").read_text(
        encoding="utf-8"
    )

    assert "+ config.api_key\n" not in source
    assert "+ config.api_key_fingerprint\n" in source
