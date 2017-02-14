test_cases = {
    "MLP::SimpleMNIST.py": {
        "dir": "Examples/Image/Classification/MLP/Python",
        "exe": "python",
        "args": ["SimpleMNIST.py"]
    },
    "MLP::01_OneHidden.cntk": {
        "dir": "Examples/Image/GettingStarted",
        "exe": "cntk",
        "args": ["configFile=01_OneHidden.cntk"]
    },
    "ResNet20::TrainResNet_CIFAR10.py": {
        "dir": "Examples/Image/Classification/ResNet/Python",
        "exe": "python",
        "args": ["TrainResNet_CIFAR10.py"]
    },
    "ResNet20::ResNet20_CIFAR10.cntk": {
        "dir": "Examples/Image/Classification/ResNet/BrainScript",
        "exe": "cntk",
        "args": ["configFile=ResNet20_CIFAR10.cntk"]
    },
    "ResNet110::TrainResNet_CIFAR10.py resnet110": {
        "dir": "Examples/Image/Classification/ResNet/Python",
        "exe": "python",
        "args": ["TrainResNet_CIFAR10.py", "-n", "resnet110"]
    },
    "ResNet110::ResNet110_CIFAR10.cntk": {
        "dir": "Examples/Image/Classification/ResNet/BrainScript",
        "exe": "cntk",
        "args": ["configFile=ResNet110_CIFAR10.cntk"]
    },
    "SLU::LanguageUnderstanding.py": {
        "dir": "Examples/LanguageUnderstanding/ATIS/Python",
        "exe": "python",
        "args": ["LanguageUnderstanding.py"]
    },
    "SLU::ATIS.cntk": {
        "dir": "Examples/LanguageUnderstanding/ATIS/BrainScript",
        "exe": "cntk",
        "args": ["configFile=ATIS.cntk"]
    },
    "S2S::Sequence2Sequence.py": {
        "dir": "Examples/SequenceToSequence/CMUDict/Python",
        "exe": "python",
        "args": ["Sequence2Sequence.py"]
    },
}