"""Fuzzy math

The fuzzy math class allows comparison operations on floats with absolute
and/or relative error bounds.

Usage
-----

    from _fuzzy import fuzzy
    fuzzy.rel_err = 0.01
    a = fuzzy(10.0)
    print(f"{a==10.01=}")
"""

class fuzzy(float):

    abs_err = None
    """Absolute error tolerance during comparisons"""

    rel_err = None
    """Relative error tolerance during comparisons"""

    def __new__(cls,*args,**kwargs):
        return super().__new__(cls,*args,**kwargs)

    def __eq__(self,x):
        delta = abs(float(self)-float(x))
        if not self.abs_err is None and delta > self.abs_err:
            return False
        if not self.rel_err is None and delta > abs(self.rel_err*float(x)):
            return False
        if self.abs_err is None and self.rel_err is None:
            return float(self) == float(x)
        return True

    def __ne__(self,x):
        return not self.__eq__(x)

    def __lt__(self,x):
        if not self.abs_err is None and float(self) > float(x) - self.abs_err:
            return False
        if not self.rel_err is None and float(self) > float(x) - abs(self.rel_err*float(x)):
            return False
        if self.abs_err is None and self.rel_err is None:
            return float(self) < float(x)
        return True

    def __le__(self,x):
        if not self.abs_err is None and float(self) > float(x) + self.abs_err:
            return False
        if not self.rel_err is None and float(self) > float(x) + abs(self.rel_err*float(x)):
            return False
        if self.abs_err is None and self.rel_err is None:
            return float(self) <= float(x)
        return True

    def __gt__(self,x):
        if not self.abs_err is None and float(self) < float(x) + self.abs_err:
            return False
        if not self.rel_err is None and float(self) < float(x) + abs(self.rel_err*float(x)):
            return False
        if self.abs_err is None and self.rel_err is None:
            return float(self) > float(x)
        return True

    def __ge__(self,x):
        if not self.abs_err is None and float(self) < float(x) - self.abs_err:
            return False
        if not self.rel_err is None and float(self) < float(x) - abs(self.rel_err*float(x)):
            return False
        if self.abs_err is None and self.rel_err is None:
            return float(self) >= float(x)
        return True


if __name__ == '__main__':
    
    a = fuzzy(10)
    rows = [(None,None),(0.01,None),(None,0.001),(0.01,0.001)]
    cols = [9.98,9.985,9.99,9.995,10.0,10.005,10.01,10.015,10.02]
    # print("abs_err rel_err"," ".join([f"{x:^7.2f}" for x in cols]))
    print(*[f"{x:^7s}" for x in ["abs_err","rel_err","a","b","a==b","a<b","a<=b","a>=b","a>b","a!=b"]])
    for fuzzy.abs_err,fuzzy.rel_err in rows:
        # print(f"\n{fuzzy.abs_err=}, {fuzzy.rel_err=}")
        print(*["------- "*10])
        x = f"{'-':s}" if fuzzy.abs_err is None else f"{fuzzy.abs_err:.3f}"
        y = f"{'-':s}" if fuzzy.rel_err is None else f"{fuzzy.rel_err:.3f}"
        for b in cols:
            print(f"{x:^7s} {y:^7s}",end=" ")
            print(f"{a:7.3f}",f"{b:7.3f}",end=" ")
            print(f"{a==b:^7d}",end=" ")
            print(f"{a<b:^7d}",end=" ")
            print(f"{a<=b:^7d}",end=" ")
            print(f"{a>=b:^7d}",end=" ")
            print(f"{a>b:^7d}",end=" ")
            print(f"{a!=b:^7d}",end=" ")
            # print("")
            # print(f"{  a  ==  b  =}".replace(" a ",str(a)).replace(" b ",str(b)))
            # print(f"{  a  !=  b  =}".replace(" a ",str(a)).replace(" b ",str(b)))
            # print(f"{  a  >   b  =}".replace(" a ",str(a)).replace(" b ",str(b)))
            # print(f"{  a  >=  b  =}".replace(" a ",str(a)).replace(" b ",str(b)))
            # print(f"{  a  <=  b  =}".replace(" a ",str(a)).replace(" b ",str(b)))
            # print(f"{  a  <   b  =}".replace(" a ",str(a)).replace(" b ",str(b)))
            print()
    print(*["------- "*10])
