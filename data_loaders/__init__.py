#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
@dataclass
class DataHandler:
    X_data: 'DataHandler' = None
    X_data_test: 'DataHandler' = None
    y_data: 'DataHandler' = None
    y_data_test: 'DataHandler' = None
data_handler = DataHandler()
