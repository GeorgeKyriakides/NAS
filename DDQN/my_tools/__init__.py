import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.realpath(sys.argv[0]))+'/my_tools')


from my_logging import ProgressLogger


def plotECDF(data, tag):
    from statsmodels.distributions.empirical_distribution import ECDF
    import matplotlib.pyplot as plt
    if data is None or data.empty:
        return plt.figure()
    ecdf = (ECDF(data))
    fig = plt.plot(ecdf.x, ecdf.y, label=(tag))
    return fig
