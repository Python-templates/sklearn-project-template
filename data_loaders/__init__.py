#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
@dataclass
class DataHanlder:
    X_data: 'DataHanlder' = None
    X_data_test: 'DataHanlder' = None
    y_data: 'DataHanlder' = None
    y_data_test: 'DataHanlder' = None
data_handler = DataHanlder()
