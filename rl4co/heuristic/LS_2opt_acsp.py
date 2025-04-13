import torch

class LS_acsp_2opt:
    def __init__(self, loc, MAXCOUNT=1000):
        self.loc = loc  # numloc, for an instance
        self.MAXCOUNT = MAXCOUNT

    def run(self):
        return self.LS_2opt(self.loc, self.bestPath, self.MAXCOUNT)

    def pathCompare(self, path1, path2): 
        if self.dist(path1) <= self.dist(path2):
            return True
        return False

    def dist(self, tour):
        loc = self.loc[tour]
        return torch.sqrt(torch.square(loc[1:]-loc[:-1]).sum(-1)).sum() + torch.sqrt(torch.square(loc[0]-loc[-1]).sum())

    def generateRandomPath(self, bestPath):

        probs = torch.ones(len(bestPath))
        probs[0] = 1e-6
        samples = torch.multinomial(probs, num_samples=2, replacement=True).to("cpu")
        a, b = torch.arange(0, len(bestPath))[samples]
        if a > b:
            return b, a, bestPath[b:a + 1]
        else:
            return a, b, bestPath[a:b + 1]


    def reversePath(self, path):
        rePath = path.clone()
        rePath[1:-1] = torch.flip(rePath[1:-1], dims=(0,))
        return rePath

    def LS_2opt(self, bestPath, MAXCOUNT=1000):
        "Swap two neighboring cities to find a better route"
        count = 0
        while count < MAXCOUNT:

            start, end, path = self.generateRandomPath(bestPath)
            # print(path)
            rePath = self.reversePath(path)
            # print(rePath)
            if self.pathCompare(path, rePath):
                count += 1
                continue
            else:
                count = 0
                bestPath[start:end + 1] = rePath
        return bestPath
