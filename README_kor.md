# CEAL_using_Tkinter

[original_README](./README.md)

## 개요

논문) [Cost-Effective Active Learning for Deep Image Classification](https://arxiv.org/abs/1701.03551)

PyTorch로 CEAL 알고리즘을 구현하고, 사용자가 실제로 선택된 불확실한 샘플에 대해 레이블링을 수행하도록 Tkinter를 활용한 GUI를 제공한다. 이를 통해 제한된 비용으로 레이블링된 데이터를 단계적으로 획득하는 Active Learning을 이해해 본다.



## 실행 방법

 실행을 위해 필요한 파라미터들은 configuration.yml 을 통해 설정한다.  파라미터 설정 후 main.py를 실행한다.

```shell
python main.py
```





*참고)*

- [CEAL_keras](https://github.com/dhaalves/CEAL_keras)
- [PyTorch implementation of CEAL](https://github.com/rafikg/CEAL)
- [Pytorch[Basics] - Sampling Samplers](https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a)