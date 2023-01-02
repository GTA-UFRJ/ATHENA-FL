#!/bin/bash

pkill -9 -f client.py
pkill -9 -f server.py
rm -rf models/*
rm results/*
