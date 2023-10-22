# simple_casadi_mpc

C++からCasADiを用いてMPCを解くためのライブラリ

# 依存関係
- CasADi([インストール用スクリプト例](https://github.com/Kotakku/OptimLibSetupHub/blob/master/CasADi/install_casadi.sh))
- Eigen3
- matplotlib(Example用)

# インストール方法
git cloneして以下を実行
```
mkdir build
cd build
cmake ..
make
sudo make install
```

# Examples
## double_integrator_mpc_example
摩擦がない物体の直動運動のようなもの  
座標1、速度1から開始して座標0に収束させる  

![](pic/simple_mpc_example.png)

## cartpole_mpc_example
cartpoleタイプの倒立振子の振り上げ  
問題設定は以下のリンクのものを仕様  

https://gist.github.com/mayataka/ef178130d52b5b06d4dd8bb2c8384c54
https://gist.github.com/mayataka/bc08faa63a94d8b48ceba77cc79c7ccc

![](pic/cartpole_mpc_example.png)


https://github.com/Kotakku/simple_casadi_mpc/assets/25411427/d4d3ffab-cfee-47f4-a080-d1ec638afa4c


## inverted_pendulum_mpc_example
回転軸にモーターがついてるタイプの倒立振子の振り上げ
トルク制約からアームのように振り上げができないように設定してあるため振り上げに助走が入る

![](pic/inverted_pendulum_mpc_example.png)


https://github.com/Kotakku/simple_casadi_mpc/assets/25411427/8fdee0d8-ef3e-4fd3-b0ae-d3c87431c2a9


## diff_drive_mpc_example
差動二輪で左上から右下まで移動する
その際に円状のオブジェクトの衝突を避けるように制約を追加している

![](pic/diff_drive_mpc_example.png)