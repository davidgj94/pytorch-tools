class Register:

	def __init__(self):
		self.registry = {}

	def attach(self, name):
		def inner(func):
			self.registry[name] = func
			return func
		return inner

	def get(self, name):
		return self.registry.get(name)