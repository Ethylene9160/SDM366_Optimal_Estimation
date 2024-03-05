import datetime
import numpy as np
import matplotlib.pyplot as plt

class CanShow:
	def __init__(self):
		pass

	def show(self):
		pass

class TimeShow(CanShow):
	def __init__(self):
		pass

	def show(self):
		current_time = datetime.datetime.now()
		print('current time:', current_time)

class ListRemove(CanShow):
	def __init__(self):
		self.mlist = ['Red' , 'Green', 'White', 'Black', 'Pink', 'Yellow']
	
	def show(self):
		self.mlist.remove('Red')
		self.mlist.remove('Pink')
		self.mlist.remove('Yellow')
		print(self.mlist)

class Student(CanShow):
	def __init__(self, name = 'default', id='default'):
		self.name = name
		self.id = id

	def show(self):
		print('name: ', self.name, ',id: ', self.id)

class NumpyUse(CanShow):
	def __init__(self):
		self.arr = np.array(range(1,11,1))

	def reshape(self):
		print('original arr:')
		print(self.arr)
		print('reshape:')
		reshaped = np.reshape(self.arr,(2,5))
		print(reshaped)
		return reshaped

	def split(self,arr):
		print('store the last 2 rows and cols:')
		print(arr[:,3:])

	def quickSort(self, arr, left, right):
		if left >= right:
			return
		pv = arr[left]
		i = left
		j = right
		while i < j:
			while i < j and arr[j] < pv:
				j -= 1
			self.swap(arr, i, j)
			while i < j and arr[i] >= pv:
				i += 1
			self.swap(arr, i, j)
		# 	if i < j:
		# 		self.swap(arr, i, j)
		# arr[left], arr[j] = arr[j], arr[left]

		self.quickSort(arr, left, j - 1)
		self.quickSort(arr, j + 1, right)

	def insert(self,arr):

		arr = arr.tolist()
		arr.insert(2,3)
		arr.append(6)
		print('after inisertion:')
		print(arr)

	def swap(self,arr, i, j):
		m = arr[i]
		arr[i] = arr[j]
		arr[j] = m

	def show(self):
		reshaped = self.reshape()
		self.split(reshaped)
		sortarr = [2, 1, 5, 3, 7, 4, 6, 8]
		self.quickSort(sortarr, 0, len(sortarr)-1)
		print('after sort:')
		print(sortarr)

		insertarr = np.array([1,2,4,5])
		self.insert(insertarr)

class MatrixCal(CanShow):
	def __init__(self):
		self.A =np.array([
			[1,-1,0],
			[1,2,2],
			[-1,0,-1],
			[0,1,0]
		])
		self.B=np.array([[-2,-1,1],[1,5,4],[1,-1,2],[1,2,1]])
		print("A:", self.A)
		print("B:", self.B)

	def cal(self):
		return self.A+self.B, self.A-self.B

	def printRowAndCol(self):
		A = self.A
		B = self.B
		print('2nd row of A:')
		print(A[1,:])
		print('3rd col of B:')
		print(B[:,2],'^T')

	def combine(self):
		# combine A and B
		return np.concatenate((self.A, self.B), axis=1)

	def dot(self):
		return np.dot(self.A.T, self.B)

	def show(self):
		self.printRowAndCol()
		a,b = self.cal()
		print('add: \n', a, '\nminus: \n', b)
		combined = self.combine()
		print('[A,B]: \n', combined)
		dot = self.dot()
		print('A^TB: \n', dot)

class mMatlabShow(CanShow):
	def __init__(self):
		super().__init__()
	def show(self):
		t = np.arange(-np.pi, np.pi, 2*np.pi/10)
		x = np.cos(t)
		y = np.sin(t)

		plt.figure()
		plt.plot(x, y, 'x')
		plt.show()



if __name__ == "__main__":
	print('===== Q1=====')
	timeshow = TimeShow()
	timeshow.show()

	listremove = ListRemove()
	listremove.show()

	stu = Student('Ganyu', '233')
	stu.show()

	print('===== Q2=====')
	mnp = NumpyUse()
	mnp.show()

	print('===== Q3=====')
	mc = MatrixCal()
	mc.show()

	print('===== Q4=====')
	mt = mMatlabShow()
	mt.show()



	
		