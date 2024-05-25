import T0_ParsingFiles as parse
import T4_3models as models


# task 5 - implement three different effectiveness measures to evaluate the document ranking
if __name__ == "__main__":
    # parse 50 queries and save in dict() collId: {query frequency}
    # queries = parse.parse_queryfile()
    # print(queries)

    coll_benchmarks = parse.evaluation_benchmark()
    print(coll_benchmarks['118'])
    