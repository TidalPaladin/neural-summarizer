#!/usr/bin/env python
from flags import parser
from train import main


if __name__ == '__main__':
    args = parser.parse_args()
    setattr(args, 'mode', 'test')
    main(args)
