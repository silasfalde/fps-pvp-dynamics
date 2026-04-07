from enum import Enum


class Terrain(str, Enum):
    OPEN = "open"
    COVER = "cover"
    CHOKE = "choke"
    SPAWN = "spawn"
    OBJECTIVE = "objective"
    WALL = "wall"


class Action(str, Enum):
    ENGAGE = "engage"
    REPOSITION = "reposition"
    PUSH_OBJECTIVE = "push_objective"
    HOLD_DEFEND = "hold_defend"
    RETREAT = "retreat"
