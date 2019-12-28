from os.path import basename, dirname, join
from glob import glob
from pathlib import Path

ROOT_MODULE = Path(__file__).parts[-2]

class Register:

	def __init__(self, register_path):
		self.registry = {}
		self.register_path = register_path
		self.current_module = Path(self.register_path).parts[-2]

	def attach(self, name):
		def inner(func):
			self.registry[name] = func
			return func
		return inner
	
	def load_modules(self,):
		pwd = dirname(self.register_path)

		for x in glob(join(pwd, '*.py')):
			if str(x) != self.register_path:
				module_name = basename(x)[:-3]
				if module_name != '__init__':
					module_name = '{}.{}.{}'.format(ROOT_MODULE, self.current_module, module_name)
					print(module_name)
					__import__(module_name, globals(), locals())

	def get(self, name):
		return self.registry.get(name)