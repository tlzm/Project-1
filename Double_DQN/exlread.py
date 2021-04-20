# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 09:49:41 2018

@author: Admin
"""

import xlrd

class OpenExl():
    #初始化exl文件，by_index表示第几个工作表单，默认值为0。
    def __init__(self, exc_file, by_index = 0):
        # 打开文件
        self.exc_data = xlrd.open_workbook(exc_file)
#        print(exc_file)
        # 获取工作表
        self.table = self.exc_data.sheets()[by_index]


    # 定义获取excle表单数据，colname_index表示列数，rowname_index表示行数，默认值都为0；
    def excel_table_data(self, colname_index=0, rowname_index=0):
        self.nrows_row = self.table.nrows  # 行数
        self.nrows_col = self.table.ncols  # 列数
        #获取整列数据
        self.exc_data2 = self.table.row_values(colname_index, rowname_index)
        return self.exc_data1,self.exc_data2


#if __name__ == '__main__':
#VCEST=OpenExl('VCEST.xlsx',1)
#bbb=VCEST.table.row_values(1)
#print(bbb[7])
#aaa,aaa1=VCEST.excel_table_data(6,0)
#print(aaa1)
##    print(aaa)
#    i = 2
#    for i in range(data_gk.table.nrows):
#        print(VCEST.table.row_values(i))
#        bbb=VCEST.table.row_values(i)
#        aaa=bbb[0]+200
#        print(aaa)
#        print(bbb[7])