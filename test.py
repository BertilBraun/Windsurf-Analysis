import modal

if __name__ == '__main__':
    import time

    yolo_model = 'windsurfing/2025_08_09_100epochs.pt'
    reid_model = 'common/osnet_ain_x1_0_msmt17.pth'

    with open('tmp/test.mp4', 'rb') as f:
        ac_bytes = f.read()

    InferenceModel = modal.Cls.from_name('windsurf-analysis', 'InferenceModel')

    start = time.time()
    for i in range(1):
        print('Sending request with', len(ac_bytes), 'bytes of video')
        print(InferenceModel().inference.spawn(f'test{i}', ac_bytes, yolo_model, reid_model, 'https://google.com'))
        time.sleep(0.5)
        print(f'Time taken: {time.time() - start} seconds')
    end = time.time()
    print(f'Time taken total: {end - start} seconds')
