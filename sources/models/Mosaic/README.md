#### Конвертация модели

`python <Директория с openvino>\deployment_tools\model_optimizer\mo.py --input_model <Путь к репозиторию>\sources\models\Mosaic\Mosaic.onnx`

#### Запуск модели

`python <Путь к репозиторию>\sources\models\Mosaic\Mosaic.py --model <Путь к репозиторию>\sources\models\Mosaic\Mosaic.xml --weights <Путь к репозиторию>\sources\models\Mosaic\Mosaic.bin --image <Путь к изображению>`