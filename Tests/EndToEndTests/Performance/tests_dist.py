test_cases = {
    "ResNet110::Distributed::TrainResNet_CIFAR10.py resnet110 2-worker": {
        "dir": "Examples/Image/Classification/ResNet/Python",
        "exe": "python",
        "args": ["TrainResNet_CIFAR10.py", "-n", "resnet110"],
        "distributed": ["-n", "2"]
    },
    "ResNet110::Distributed::ResNet110_CIFAR10.cntk 2-worker": {
        "dir": "Examples/Image/Classification/ResNet/BrainScript",
        "exe": "cntk",
        "args": ["configFile=ResNet110_CIFAR10.cntk"],
        "distributed": ["-n", "2"]
    }
}
