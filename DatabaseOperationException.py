class DatabaseOperationException(IndexError):
    def __init__(self,*args,**kwargs):
        super().__init__(self,*args,**kwargs)