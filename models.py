from model import *
import cv2


def process_frame42_downsampled(frame):
    """
    Process the Atari frame to a 42*42 images in grayscale downsampled to 3 bits colors (as in A3C+ paper)
    :param frame: the Atari frame, of shape (210, 160, 3)
    :return: the processed frame
    """

    # Select interesting area of size 160*160*3
    frame = frame[34:34 + 160, :160, :]
    # Convert to gray scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    # downsampled the image
    frame = frame.astype(np.float32)
    frame = np.uint8(frame / 255 * 8)  # TODO 8 in the paper, buy maybe try another value (128 ?)

    # Final reshape
    frame = np.reshape(frame, [42, 42, 1])
    return frame


def process_frame42_v2(frame):
    """
    Process the Atari frame to a 42*42 images in grayscale downsampled to 3 bits colors (as in A3C+ paper)
    :param frame: the Atari frame, of shape (210, 160, 3)
    :return: the processed frame
    """

    # Select interesting area of size 160*160*3
    frame = frame[34:34 + 160, :160, :]
    # Convert to gray scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    # downsampled the image
    frame = frame.astype(np.uint8)
    frame *= 8. / 255.  # TODO 8 in the paper, buy maybe try another value (128 ?)

    # Final
    frame = frame.astype(np.float32)
    frame = np.uint8(frame / 8)
    return frame


def process_frame42_pos(frame):
    # print(frame.shape) # (210, 160, 3)
    # Convert to gray scale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Select interesting area
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame


def process_frame84(frame):
    # print(frame.shape) # (210, 160, 3)
    # Convert to gray scale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Select interesting area
    frame = frame[34:34 + 160, :160]
    # Resize  down to 84x84
    frame = cv2.resize(frame, (84, 84))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (2.0 / 255.0)
    frame -= 1.0
    frame = np.reshape(frame, [84, 84, 1])
    return frame


possible_model = ['LSTM', 'VIN', 'FF', 'FFWider', 'FFDeeper', 'FFWiderAndDeeper', 'VINDeeperCNN', 'DeepMind', 'VIN2D']
one_input_brain = ['LSTM']  # List of the brain that use only one input frame. The rest of the input are RNN data

model_name_to_class = {
    'LSTM': LSTMPolicy,
    'VIN': VINPolicy,
    'VINDeeperCNN': VINDeeperCNNPolicy,
    'FF': FFPolicy,
    'DeepMind': DeepMindPolicy,
    'FFWider': FFWiderPolicy,
    'FFDeeper': FFDeeperPolicy,
    'FFWiderAndDeeper': FFWiderAndDeeperPolicy,
    'VIN2D': VIN2DPolicy
}

model_name_to_process = {
    'LSTM': process_frame42_pos,
    'VIN': process_frame42_pos,
    'VINDeeperCNN': process_frame42_pos,
    'FF': process_frame42_pos,
    'FFWider': process_frame42_pos,
    'FFDeeper': process_frame42_pos,
    'FFWiderAndDeeper': process_frame42_pos,
    'DeepMind': process_frame84,
    'VIN2D': process_frame42_pos
}
