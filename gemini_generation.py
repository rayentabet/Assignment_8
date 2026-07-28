import threading
import time


MINIMUM_INTERVAL = 4.5
lock = threading.Lock()
last_request_time = 0


def generate(client, model, contents):
    global last_request_time

    with lock:
        elapsed = time.monotonic() - last_request_time
        if elapsed < MINIMUM_INTERVAL:
            time.sleep(MINIMUM_INTERVAL - elapsed)

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
        except Exception as error:
            if "429" not in str(error):
                raise

            time.sleep(30)
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )

        last_request_time = time.monotonic()
        return response
