from grafico import graffic
from londresCriminal import cleanData
from DBSCAN import applyDBSCAN


if __name__ == "__main__":
    data=cleanData()
    graffic(data)
    applyDBSCAN(data)
