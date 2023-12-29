import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from datetime import datetime
from src.config import DATA_DIR, LOG_FILE_PATH
from src.learning_to_rank import train, utils
from src.manual_ranking.typings import EmbedderConfig


class MyHandler(PatternMatchingEventHandler):
    def __init__(self, log_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file

    def on_modified(self, event):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp}\n"
        train.main(
            file_path=event.src_path,
            debug=False,
            embbeder_config=EmbedderConfig(name="albert-base-v2", type="hf"),
        )  # TODO : Option to change the config file dynamically
        with open(self.log_file, "a") as log:
            log.write(log_entry)


def watch():
    observer = Observer()
    event_handler = MyHandler(
        log_file=LOG_FILE_PATH,
        patterns=["*.csv"],
        ignore_directories=True,
    )
    observer.schedule(event_handler, path=DATA_DIR, recursive=False)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
