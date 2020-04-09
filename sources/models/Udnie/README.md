#### Конвертация модели

`python <Директория с openvino>\deployment_tools\model_optimizer\mo.py --input_model <Путь к репозиторию>\sources\models\Udnie\Udnie.onnx`

#### Запуск модели

`python <Путь к репозиторию>\sources\models\Udnie\Udnie.py --model <Путь к репозиторию>\sources\models\Udnie\Udnie.xml --weights <Путь к репозиторию>\sources\models\Udnie\Udnie.bin --image <Путь к изображению>`