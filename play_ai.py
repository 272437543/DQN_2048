from game_2048 import Game2048
import numpy as np
from keras.models import load_model
import time

model = load_model('pretrained/dqn2048_cnn.h5')
model.summary()
game = Game2048()
game.reset()
s = game.get_state()
game_step = 0
while True:
    print(game.board)
    state = s[np.newaxis, :, :, np.newaxis]
    state = np.log(state + 1) / 16
    action_index = model.predict(state)

    s_, r, done = game.step(np.argmax(action_index))
    # print('action:', game.actions[action_index])
    # print('game:\n', s_, '\n')

    s = s_
    if done:
        print('final:\n', game.board)
        print('score:', game.get_score(), ' board sum:', np.sum(game.board), ' play step:', game.n_step)
        break

    game_step += 1
    time.sleep(1)
