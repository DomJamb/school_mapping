import requests
from PIL import Image
from io import BytesIO
import threading
from queue import Queue
import time

def producer(task_queue, tasks):
    for task in tasks:
        task_queue.put(task)

def consumer(task_queue):
    while True:
        task = task_queue.get()
        if task is None:  # Stop signal
            task_queue.task_done()
            break
        image_name, bbox = task
        image = fetch_satellite_image(bbox)
        if isinstance(image, Image.Image):
            save_image(image, image_name)
            print(f"{image_name} saved.")
        else:
            print(f"Failed to download image for {bbox}: {image}")
        task_queue.task_done()

def fetch_satellite_image(bbox):
    url_template = (
        "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export?"
        "bbox={bbox}&bboxSR=4326&imageSR=4326&size=500,500&format=jpeg&f=image"
    )
    url = url_template.format(bbox=",".join(map(str, bbox)))
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return f"Error: {response.status_code}"

def save_image(img, output_path):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(output_path)

# --- Example usage ----
if __name__ == '__main__':
    # BBOX format: (longitude_min, latitude_min, longitude_max, latitude_max)
    # -- i.e. (west boundary, south boundary, east boundary, north boundary)
    start_bbox = (15.9487733526612, 45.79284504132088, 15.952654512188677, 45.79556888808291)

    # generate 1000 bboxes from the start_bbox with shift of 0.001
    bbox_id_list = []
    for i in range(100):
        for j in range(10):
            bbox = (start_bbox[0] + i * 0.001, start_bbox[1] + j * 0.001, start_bbox[2] + i * 0.001, start_bbox[3] + j * 0.001)
            image_id = f"image_{i}_{j}"
            bbox_id_list.append((image_id, bbox))

    dest_dir = 'satellite_imgs'

    # each task for a consumer thread is a tuple: (dest_image_path, bbox)
    tasks = [
        (f"{dest_dir}/{image_id}.jpeg", bbox) for image_id, bbox in bbox_id_list
    ]
    task_queue = Queue()

    # Number of consumer threads
    num_consumer_threads = 100 

    start = time.time()

    # Start producer thread
    producer_thread = threading.Thread(target=producer, args=(task_queue, tasks))
    producer_thread.start()

    # Start consumer threads
    consumer_threads = [
        threading.Thread(target=consumer, args=(task_queue,)) for _ in range(num_consumer_threads)
    ]
    for thread in consumer_threads:
        thread.start()

    # Wait for all tasks to be processed
    producer_thread.join()  # Ensure producer finishes enqueueing all tasks

    for _ in range(num_consumer_threads):
        task_queue.put(None)

    # Wait for all consumer threads to complete
    for thread in consumer_threads:
        thread.join()

    end = time.time()

    print('Total time for 1000 images:', end - start)
    print('Time per image:', (end - start) / 1000)
