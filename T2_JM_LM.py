import math, os
import T0_ParsingFiles as parse

if __name__ == "__main__":
    # parse 50 queries and save in dict() collId: {query frequency}
    queries = parse.parse_queryfile()