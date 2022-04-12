# Multi-agent Communication Algs

Implemented based on [Tianshou](https://github.com/thu-ml/tianshou).

## Algs
- [ ] DDPG & SAC
- [ ] MADDPG & MASAC
- [ ] [DIAL](https://arxiv.org/abs/1605.06676)
- [ ] [CommNet](https://arxiv.org/abs/1605.07736)
- [ ] [TarMAC](https://arxiv.org/abs/1810.11187)
- [ ] [I2C](http://arxiv.org/abs/2006.06455)
- [ ] [Intention Sharing](https://openreview.net/forum?id=qpsl2dR9twy)

Recommend [A Survey of Multi-Agent Reinforcement Learning with Communication](http://arxiv.org/abs/2203.08975) for a detailed taxonomy.

## Training Schemes
| Types | Sub-types |
| -- | -- |
| Fully Decentralized | |
| CTDE | Individual Parameter |
| | Parameter Sharing |
| | Individual Parameter with Global Info |


## Logic of Tianshou in MARL

```mermaid
flowchart LR
    subgraph mapolicy [MA-policy]
        p1((Agent1)) --action--> m([Manager]) -->|obs or messages| p1
        p2((Agent2)) --action--> m([Manager]) -->|obs or messages| p2
        p3((Agent3)) --action--> m([Manager]) -->|obs or messages| p3
    end    
    
    subgraph collector [Collector]
        VE(VecEnv) ==Transition==> mapolicy ==Action==> VE;
    end
    
    subgraph alg [Algorithm]
        collector ==Data==> B[(Buffer)] ==Sample==> T{Trainer} ==>|Processed Sample| mapolicy ==Info==> T
        T ==Info==> L{{Logger}}
    end
    

```

An algorithm corresponds to 
+ A MA-policy: interaction among agents, such as the communication 
+ A Buffer: what to store
+ A Trainer: Update, but implemented in each agent's policy actually

## Instructions

### Install
```shell
sudo apt install swig -y
pip install tianshou 'pettingzoo[all]'
```