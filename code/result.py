class Result():
    def __init__(self):
        self.type = ""
        self.val = ""
        self.bbox = []
        
    def set_type(self,type):
        self.type = type
        
    def set_val(self, val):
        self.val = val
        
    def append_val(self, val):
        self.val += val
        
    def set_bbox(self, bbox):
        self.bbox = bbox
        
    def set_image_path(self, image_path):
        self.image_path = image_path
    