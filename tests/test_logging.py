from __future__ import annotations

import logging

from personalityrag.logger import (
    LOGGER_NAME,
    WebLogBuffer,
    configure_logging,
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

    assert log_file.exists()
    assert (tmp_path / "logs" / "personalityrag.log.1").exists()
    assert get_log_buffer().summary()["entry_count"] <= 20


def test_log_sanitizer_redacts_secrets():
    text = sanitize_log_message(
        "api_key=secret authorization=BearerToken prag_abcdefghijklmnopqrstuvwxyz"
    )
    assert "secret" not in text
    assert "BearerToken" not in text
    assert "prag_abcdefghijklmnopqrstuvwxyz" not in text
    assert "[redacted]" in text
