import gym
from keras.models import load_model
import numpy as np

#0: Rien, 1: Gauche, 2: Bas, 3: Droite
env = gym.make('LunarLander-v2')
model = load_model('LunarLander-v22.h5')

env.seed = 10
state = env.reset()
done = False


while not done:
	action = np.argmax(model.predict(state.reshape((1, -1))))
	state, reward, done, _ = env.step(action)
	env.render()

env.close()