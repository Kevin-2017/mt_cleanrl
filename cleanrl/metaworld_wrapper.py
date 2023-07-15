import metaworld
import random
import gymnasium as gym
def load_mt(environment_name,
            env_id=None, render_mode = None,  seed = 0):
    if environment_name == 'mt1-pick':
        environment_name_list = ['pick-place-v2']
    else:
        environment_name_list = [environment_name]
    #TODO(Kevin) Add more environments in the future

    envs = []
    full_env_name_list = []
    exist_duplicate = len(environment_name_list) != len(
        set(environment_name_list))
    for i, env_name in enumerate(environment_name_list):
        mt1 = metaworld.MT1(env_name, seed=seed)
        env = mt1.train_classes[env_name]()
        # task = mt.train_tasks[0]
        task = random.choice(mt1.train_tasks)
        env.set_task(task)

        if exist_duplicate:
            full_env_name = str(i) + '-' + env_name
        else:
            full_env_name = env_name
        full_env_name_list.append(full_env_name)
        # print(env)
        # env = gym_wrapper(
        #     env,
        #     env_id=env_id,
        #     render_mode=render_mode
        # )
        envs.append(env)
    return envs
    
def load_mt_benchmark(environment_name,
                      env_id=None,
                      discount=1.0,render_mode = None,  seed = 0):
    assert environment_name in ['mt10', 'mt50']
    if environment_name == 'mt10':
        mt_benchmark = metaworld.MT10(seed=seed)
    else:
        mt_benchmark = metaworld.MT50(seed=seed)


if __name__ == "__main__":
    # unit test
    env = load_mt("pick-place-v2")[0]
    env.set_render('human')
    obs = env.reset()  # Reset environment
    done = False

    while not done:
        # Perform a random action
        action = env.action_space.sample()
        # Step through the environment
        next_obs, rewards, terminateds, truncateds, infos = env.step(action)
        # print(action)
        # Visualize the environment
        env.render()

