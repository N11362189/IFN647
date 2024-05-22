import T0_ParsingFiles as parse
import T4_3models as models


# task 5 - compare the 3 models
if __name__ == "__main__":
    # parse 50 queries and save in dict() collId: {query frequency}
    queries = parse.parse_queryfile()
    # print(queries)