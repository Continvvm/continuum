

def require_subset(subset):
    def wrapper1(func):
        def wrapper2(self):
            if subset not in self.list_subsets:
                raise Exception(
                    f"No {subset} predictions have been logged so far which "
                    f"{func.__name__} rely on!"
                )
            return func(self)
        return wrapper2
    return wrapper1


def cache(func):
    def wrapper(self):
        name = f"__cached_{func.__name__}"
        v = self.__dict__.get(name)
        if v is None:
            v = func(self)
            self.__dict__[name] = v
        return v
    return wrapper
