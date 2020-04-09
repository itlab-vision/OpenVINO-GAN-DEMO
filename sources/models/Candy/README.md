#### Конвертация модели

`python <Директория с openvino>\deployment_tools\model_optimizer\mo.py --input_model <Путь к репозиторию>\sources\models\Candy\Candy.onnx`

#### Запуск модели

`python <Путь к репозиторию>\sources\models\Candy\candy.py --model <Путь к репозиторию>\sources\models\Candy\candy.xml --weights <Путь к репозиторию>\sources\models\Candy\candy.bin --image <Путь к изображению>`