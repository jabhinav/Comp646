# Imitation Learning v/s Reinforcement Learning in Grid Worlds

## Sample 10x10 Grid World with Obstacles
Task: Collect Marker and Bring it to the Goal
![world](https://user-images.githubusercontent.com/19610318/233808029-d9f88a99-cd08-4080-bd4a-0688ee643737.png)

## Visualisation of the Expert Policy
https://user-images.githubusercontent.com/19610318/233808198-f87970fd-4173-4066-8fab-5b24ab603564.mov

## Installation
```
# Clone the repository
git clone <repo_url>

# Install the requirements
pip install -r requirements.txt
```

## Project Structure
```
main.py: Main script to run the experiments (includes expert policy and DQN)
config.py: Contains the hyperparameters for the experiments
domain.py: Contains the environment class for Karel Grid World
networks.py: Contains the neural network architecture for DQN (MLP and CNN)
expertPolicy.py: Contains the expert policy for Karel Grid World
utils.py: Contains the utility functions for the project
```

## Usage
```
# Change the hyperparameters in config.py (comments specify the suggested values)

# Run the script to learn the policy using DQN
python main.py
```

## Results

For full-explanation of the results, refer to the included COMP_646_Final_Report.pdf

### Our best learned Policy in Action:

https://user-images.githubusercontent.com/19610318/234846281-114276e0-ce66-4455-9a7b-f3f661fffe82.mov


https://user-images.githubusercontent.com/19610318/234846298-faa06f62-be55-4ec5-950c-fe4c6676a6c8.mov

### Fail Cases where Agent gets stuck in Oscillating behaviors

https://user-images.githubusercontent.com/19610318/234846444-53fec4d7-61a8-4088-b06d-bf9c90660e71.mov

### Plots of success rate and average episode length 
![3x3random_eplen](https://user-images.githubusercontent.com/19610318/234846686-38f26a8a-930d-4366-afa4-d84a3a58ff1a.png)
![3x3random_success](https://user-images.githubusercontent.com/19610318/234846689-dfdc472b-0d1e-4daa-8c45-81c8aaaecb41.png)
![5x5random_eplen](https://user-images.githubusercontent.com/19610318/234846690-0f49ae5f-f458-4056-90c0-bf577969e89d.png)
![5x5random_success](https://user-images.githubusercontent.com/19610318/234846691-a4ef6be3-815c-4e04-9ca5-af2323d31e52.png)



