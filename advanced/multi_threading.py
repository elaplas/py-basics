from multiprocessing import Pool, cpu_count
from command_line_execution import run_app


############### data parallelism ############
# Run the same task on the different data in parallel

def multiply(x, y):
    return x*y

def calc():
    """Calculates the multiplication of the lists of two numbers in parallel 
    """
    X = [i for i in range(1000)]
    Y = [i for i in range(1000)]
    args = zip(X, Y)
    s = cpu_count()
    with(Pool(cpu_count()-3) as pool):
        results = pool.starmap(multiply, args)
    print(results[:11])

def multi_runs():
    """Runs an executable multiple times in parallel on different data/args
    """
    args = ["trivial command lines" for i in range(100)]
    with(Pool(cpu_count()-3) as pool):
        pool.starmap(run_app, args)


if __name__ == '__main__':
    calc()