from iris.models.pooling import MeanMaxPoolMIL
from iris.models.abmil import GatedAttentionMIL

MODELS = {"meanmax": MeanMaxPoolMIL, "abmil": GatedAttentionMIL}
