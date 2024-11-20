class Result():
    def __init__(self):
        self.type = ""
        self.val = ""
        
    def set_type(self,type):
        self.type = type
        
    def set_val(self, val):
        self.val = val
        
    def append_val(self, val):
        self.val += val