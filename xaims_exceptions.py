#############################################
# Xaims Exceptions
#############################################


class AimsQueued(Exception):
    def __init__(self, msg='Queued', cwd=None):
        self.msg = msg
        self.cwd = cwd

    def __str__(self):
        return repr(self.cwd)


class AimsSubmitted(Exception):
    def __init__(self, jobid):
        self.jobid = jobid

    def __str__(self):
        return repr(self.jobid)


class AimsRunning(Exception):
    pass


class AimsNotFinished(Exception):
    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return self.message


class AimsNotConverged(Exception):
    pass


class AimsUnknownState(Exception):
    pass


class AimsWarning(Exception):
    '''Exception class for Aims warnings that cause problems in xaims.'''
    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return self.message
