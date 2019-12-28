from os.path import basename, dirname, join
from glob import glob

class Register:

	def __init__(self, register_path):
		self.registry = {}
		self.register_path = register_path

	def attach(self, name):
		def inner(func):
			self.registry[name] = func
			return func
		return inner
	
	def load_modules(self,):
		pwd = dirname(self.register_path)
		for x in glob(join(pwd, '*.py')):
			if not x.startswith('__') and str(x) != self.register_path:
				__import__(basename(x)[:-3], globals(), locals())

	def get(self, name):
		return self.registry.get(name)