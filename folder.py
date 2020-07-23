




class StandardModule:
    def __init__(self, starting_path, local_path):

        self.path = starting_path+local_path

        self.input = self.path+'input/'
        self.output = self.path+'output/'


class DataScienceModule:
    def __init__(self, starting_path, datascience_path, local_path):
        self.path = starting_path+datascience_path+local_path
        self.input = starting_path+datascience_path+local_path+'input/'
        self.output = starting_path+datascience_path+local_path+'output/'



class Folder:
    def __init__(self, **kwargs):
        starting_path = '../' if 'starting_path' not in kwargs else kwargs['starting_path']
        datascience_path = 'DataScience/' if 'datascience_path' not in kwargs else kwargs['datascience_path']
        analysis_path = 'Analysis/' if 'analysis_path' not in kwargs else kwargs['analysis_path']

        self.Shakespeare = DataScienceModule(starting_path, datascience_path, 'Shakespeare/')
        self.WikiCrawl = DataScienceModule(starting_path, datascience_path, 'WikiCrawl/')
        self.ElementsStatData = DataScienceModule(starting_path, datascience_path, 'ElementsStatData/')
        self.playground = StandardModule(starting_path, '')
        self.Disneyland = DataScienceModule(starting_path, datascience_path, 'Disneyland/')
        self.watermark = DataScienceModule(starting_path, datascience_path, 'watermark/')

        self.PDC = StandardModule(starting_path, 'PDC/')

