import logging

""" A formatter class for python's logging modle that splits multiline
messages to several lines, without printing the header (timestamp and
all that) again, but indenting with the width of the header.
"""

class MultilineFormatter(logging.Formatter):
    def format(self, record):
        string = logging.Formatter.format(self, record)
        header, footer = string.split(record.message)
        string = string.replace('\n', '\n' + ' '*len(header))
        return string

