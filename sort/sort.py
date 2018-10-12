
import numpy as np

def bubble_sort(arr):
    '''
    冒泡排序
    :param arr:
    :return:
    '''
    for i in range(0,len(arr)-1):
        for j in range(1,len(arr)):
            if arr[j-1]>arr[j]:
                arr[j-1],arr[j]=arr[j],arr[j-1]
    return arr

def select_sort(arr):
    '''
    快速排序
    :param arr:
    :return:
    '''
    for i in range(0,len(arr)-1):
        index=i
        for j in range(i,len(arr)):
            if arr[j]<arr[index]:
                index=j
        if i!=index:
            arr[i],arr[index]=arr[index],arr[i]


def insert_sort(arr):
    '''
    插入排序
    :param arr:
    :return:
    '''
    for i in range(1,len(arr)):
        current=arr[i]
        j=i-1
        while(j>=0 and arr[j]>current):
            arr[j+1]=arr[j]
            j-=1
        arr[j+1]=current

def shell_sort(arr):
    '''
    shell 排序 n**2
    :param arr:
    :return:
    '''
    gap=1
    while gap<len(arr)/3:
        gap=gap*3+1
    while gap>0:
        for i in range(1, len(arr),gap):
            current = arr[i]
            j=i-gap
            while (j >= 0 and arr[j] > current):
                arr[j+gap] = arr[j]
                j -= gap
            arr[j+gap] = current
        gap=int(np.floor(gap/3))


def quicksort(arr):
    '''
    非严格意义的快排
    自顶向下
    nlogn
    :param arr:
    :return:
    '''
    if len(arr) <= 1:
        return arr
    pivot = arr[int(len(arr) / 2)]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def mergeSort(arr):
    '''
    自下而上，分而治之 nlogn
    :param arr:
    :return:
    '''
    def merge(left, right):
        result = []
        while left and right:
            if left[0] <= right[0]:
                result.append(left.pop(0));
            else:
                result.append(right.pop(0));
        while left:
            result.append(left.pop(0));
        while right:
            result.append(right.pop(0));
        return result
    if(len(arr)<2):
        return arr
    middle = int(np.floor(len(arr)/2))
    left, right = arr[0:middle], arr[middle:]
    return merge(mergeSort(left), mergeSort(right))

def heap_sort(arr):
    '''
    堆排序 nlogn
    :param arr:
    :return:
    '''
    def heap_ajust(arr,parent,end):

        temp=arr[parent]
        left_child=parent*2+1
        while left_child<=end:
            if left_child<end and arr[left_child]<arr[left_child+1]:
                left_child+=1

            if arr[left_child]<temp:
                break
            arr[parent]=arr[left_child]
            parent=left_child
            left_child=left_child*2+1
        arr[parent]=temp

    # 构建大顶堆
    for i in range(int(len(arr)/2),-1,-1):
        heap_ajust(arr,i,len(arr)-1)

    # 交换，重构
    for i in range(len(arr)-1,-1,-1):
        arr[0],arr[i]=arr[i],arr[0]
        heap_ajust(arr,0,i-1)

def count_sort(arr):
    '''
    计数排序
    :param arr:
    :return:
    '''
    buket=[0]*(np.max(arr)+1)

    for i in arr:
        buket[i]+=1

    arr=[]
    for i in range(len(buket)):
        if buket[i]!=0:
            arr+=[i]*buket[i]
    return arr

def radix_sort(arr):
    '''
    基数排序
    :param arr:
    :return:
    '''
    time=len(str(np.max(arr)))
    for i in range(time):
        s = [[] for x in range(10)]
        for j in arr:
            s[j // (10 ** i) % 10].append(j)
        arr=[x for t in s for x in t]
    return arr

def buket_sort(arr):
    '''
    桶排序
    :param arr:
    :return:
    '''
    max=np.max(arr)
    min=np.min(arr)
    buket=[0]*(max-min+1)
    for i in arr:
        buket[i-min]+=1
    arr=[]
    for i in range(len(buket)):
        if 0!=buket[i]:
            arr+=[i+min]*buket[i]

    return arr



arr=[1,4, 55, 35 ,89,45,12,12]
arr=buket_sort(arr)
# arr=mergeSort(arr)
print(arr)