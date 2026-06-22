from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    ZOOM_IN = 4
    ZOOM_OUT = 5
    STOP = 6
    UP_LEFT = 7
    UP_RIGHT = 8
    DOWN_LEFT = 9
    DOWN_RIGHT = 10


ACTION_NAMES = {
    Action.LEFT: "left",
    Action.RIGHT: "right",
    Action.UP: "up",
    Action.DOWN: "down",
    Action.ZOOM_IN: "zoom_in",
    Action.ZOOM_OUT: "zoom_out",
    Action.STOP: "stop",
    Action.UP_LEFT: "up_left",
    Action.UP_RIGHT: "up_right",
    Action.DOWN_LEFT: "down_left",
    Action.DOWN_RIGHT: "down_right",
}

NUM_ACTIONS = len(Action)
