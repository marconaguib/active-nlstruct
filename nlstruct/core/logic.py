import contextlib

import numpy as np
import pandas as pd
import pyeda.boolalg
import pyeda.boolalg.expr
import pyeda.inter
import z3
from pyeda.boolalg.minimization import espresso_exprs
from scipy.sparse import csr_matrix

from nlstruct.core.cache import hash_object
from nlstruct.core.pandas import df_to_csr, csr_to_df, factorize_rows


class LabelSubspace(object):
    def __init__(self, *children):
        assert all(isinstance(c, LabelSubspace) for c in children)
        self.children = children

    def __eq__(self, b):
        return Equivalent(self, b)

    def __ne__(self, b):
        return Not(Equivalent(self, b))

    def __and__(self, b):
        return And(self, b)

    __rand__ = __and__

    def __or__(self, b):
        return Or(self, b)

    __ror__ = __or__

    def __lshift__(self, b):
        return Implies(b, self)

    def __rshift__(self, b):
        return Implies(self, b)

    __rrshift__ = __lshift__
    __rlshift__ = __rshift__

    def __invert__(self):
        return Not(self)

    def __hash__(self):
        return hash(hash_object(self))

    def __repr__(self):
        return self.to_python_string()

    def any_factory(self):
        if hasattr(self, 'factory'):
            return self.factory
        for c in self.children:
            factory = c.any_factory()
            if factory:
                return factory

    def to_pyeda(self):
        return self.to_pyeda_()

    def to_z3(self):
        return self.to_z3_()

    def to_cnf(self, factory=None, simplify=True):
        factory = factory or self.any_factory()
        res = factory.from_pyeda(self.to_pyeda().to_cnf())
        if not isinstance(res, And):
            if not isinstance(res, Or):
                return And(Or(res))
            return And(res)
        return res

    def to_dnf(self, factory=None, simplify=True):
        factory = factory or self.any_factory()
        res = factory.from_pyeda(self.to_pyeda().to_dnf())
        if not isinstance(res, Or):
            if not isinstance(res, And):
                return Or(And(res))
            return Or(res)
        return res

    def satisfy_all(self, factory=None, lib="z3"):
        if lib == "z3":
            solver = z3.Solver()
            solver.add(self.to_z3())
            res = solver.check()
            return res == z3.sat
        elif lib == "pyeda":
            factory = factory or self.any_factory()
            all_res = self.to_pyeda().satisfy_all()
            return ({factory[c.name]: v for c, v in res.items()}
                    for res in all_res)
        else:
            raise Exception()

    def to_python_string(self):
        return self.to_python_()

    def vectorize(self):
        expr = self.vectorize_()
        return eval("lambda c: {}".format(expr))

    def to_python(self):
        expr = self.to_python_string()
        return eval("lambda c: {}".format(expr))

    def vectorize_(self):
        raise NotImplementedError()

    def to_python_(self):
        raise NotImplementedError()

    def to_pyeda_(self):
        raise NotImplementedError()

    def to_z3_(self):
        raise NotImplementedError()

    @property
    def support(self):
        res = set()
        for c in self.children:
            res |= c.support
        return res

    @classmethod
    def from_pyeda(cls, expr, factory):
        if isinstance(expr, pyeda.boolalg.expr.AndOp):
            return And(*(cls.from_pyeda(c, factory) for c in expr.xs))
        elif isinstance(expr, pyeda.boolalg.expr.OrOp):
            return Or(*(cls.from_pyeda(c, factory) for c in expr.xs))
        elif isinstance(expr, pyeda.boolalg.expr.NotOp):
            return Not(*(cls.from_pyeda(c, factory) for c in expr.xs))
        elif isinstance(expr, pyeda.boolalg.expr.Complement):
            return Not(cls.from_pyeda(expr.__invert__(), factory))
        elif isinstance(expr, pyeda.boolalg.expr.Variable):
            return factory[expr.name]
        elif isinstance(expr, pyeda.boolalg.expr.Zero):
            return empty
        elif isinstance(expr, pyeda.boolalg.expr.One):
            return full_space
        else:
            raise Exception(f"Unrecognized pyeda object {type(expr)}")


class And(LabelSubspace):

    def to_pyeda_(self):
        if len(self.children) > 1:
            return pyeda.inter.And(*(c.to_pyeda_() for c in self.children))
        else:
            return self.children[0].to_pyeda_()

    def to_z3_(self):
        if len(self.children) > 1:
            return z3.And(*(c.to_z3_() for c in self.children))
        else:
            return self.children[0].to_z3_()

    def to_python_(self):
        if len(self.children) > 2:
            return "(all(({})))".format(", ".join(c.to_python_() for c in self.children))
        elif len(self.children) == 2:
            return "({} & {})".format(self.children[0].to_python_(), self.children[1].to_python_())
        else:
            return self.children[0].to_python_()

    def vectorize_(self):
        if len(self.children) > 2:
            return "(np.all(({}), axis=0))".format(", ".join(c.vectorize_() for c in self.children))
        elif len(self.children) == 2:
            return "({} & {})".format(self.children[0].vectorize_(), self.children[1].vectorize_())
        else:
            return self.children[0].vectorize_()


class Or(LabelSubspace):

    def to_pyeda_(self):
        if len(self.children) > 1:
            return pyeda.inter.Or(*(c.to_pyeda_() for c in self.children))
        else:
            return self.children[0].to_pyeda_()

    def to_z3_(self):
        if len(self.children) > 1:
            return z3.Or(*(c.to_z3_() for c in self.children))
        else:
            return self.children[0].to_z3_()

    def to_python_(self):
        if len(self.children) > 2:
            return "(any(({})))".format(", ".join(c.to_python_() for c in self.children))
        elif len(self.children) == 2:
            return "({} | {})".format(self.children[0].to_python_(), self.children[1].to_python_())
        else:
            return self.children[0].to_python_()

    def vectorize_(self):
        if len(self.children) > 2:
            return "(np.any(({}), axis=0))".format(", ".join(c.vectorize_() for c in self.children))
        elif len(self.children) == 2:
            return "({} | {})".format(self.children[0].vectorize_(), self.children[1].vectorize_())
        else:
            return self.children[0].vectorize_()


class Not(LabelSubspace):

    def to_pyeda_(self):
        return pyeda.inter.Not(*(c.to_pyeda_() for c in self.children))

    def to_z3_(self):
        return z3.Not(*(c.to_z3_() for c in self.children))

    def to_python_(self):
        return "(~{})".format(self.children[0].to_python_())

    def vectorize_(self):
        return "(~{})".format(self.children[0].vectorize_())


class Implies(LabelSubspace):

    def to_pyeda_(self):
        return pyeda.inter.Implies(*(c.to_pyeda_() for c in self.children))

    def to_z3_(self):
        return z3.Implies(*(c.to_z3_() for c in self.children))

    def to_python_(self):
        return "(~{} | {})".format(self.children[0].to_python_(), self.children[1].to_python_())

    def vectorize_(self):
        return "(~{} | {})".format(self.children[0].vectorize_(), self.children[1].vectorize_())


class Equivalent(LabelSubspace):

    def to_pyeda_(self):
        return pyeda.inter.Equal(*(c.to_pyeda_() for c in self.children))

    def to_z3_(self):
        return self.children[0].to_z3_() == self.children[1].to_z3_()

    def to_python_(self):
        return "({} == {})".format(self.children[0].to_python_(), self.children[1].to_python_())

    def vectorize_(self):
        return "({} == {})".format(self.children[0].vectorize_(), self.children[1].vectorize_())


class ITE(LabelSubspace):
    def to_pyeda_(self):
        return pyeda.inter.ITE(*(c.to_pyeda_() for c in self.children))

    def to_z3_(self):
        return z3.If(self.children[0].to_z3_(), self.children[1].to_z3_(), self.children[2].to_z3_())

    def to_python_(self):
        return "(({} & {}) | (~{} & {}))".format(self.children[0].to_python_(),
                                                 self.children[1].to_python_(),
                                                 self.children[0].to_python_(),
                                                 self.children[2].to_python_())

    def vectorize_(self):
        return "(({} & {}) | (~{} & {}))".format(self.children[0].vectorize_(),
                                                 self.children[1].vectorize_(),
                                                 self.children[0].vectorize_(),
                                                 self.children[2].vectorize_())


class AtomLabelSubspace(LabelSubspace):
    def __init__(self, uid, index, factory):
        self.index = index
        self.factory = factory
        self.uid = str(uid)
        super().__init__()

    def __hash__(self):
        return hash(self.uid)

    @property
    def absolute_idx(self):
        return self.index

    @property
    def support(self):
        return {self}

    @property
    def name(self):
        return self.uid

    def to_pyeda_(self):
        return pyeda.inter.exprvar(self.uid)

    def to_python_(self):
        return f"c.get('{self.uid}', False)"

    def to_z3_(self):
        return z3.Bool(self.uid)

    def vectorize_(self):
        return f"c[:, {self.index}]"


class ExactlyOne(LabelSubspace):
    def to_pyeda_(self):
        return pyeda.inter.OneHot(*[a.to_pyeda_() for a in self.children])

    def to_python_(self):
        return "(sum(({})) == 1)".format(", ".join(c.to_python_() for c in self.children))

    def to_z3_(self):
        return z3.PbEq(*[a.to_z3_() for a in self.children], 1)

    def vectorize_(self):
        return "(np.sum(({}), axis=0) == 1)".format(", ".join(c.vectorize_() for c in self.children))


class AtMostOne(LabelSubspace):
    def to_pyeda_(self):
        return pyeda.inter.OneHot0(*[a.to_pyeda_() for a in self.children])

    def to_python_(self):
        return "(sum(({})) <= 1)".format(", ".join(c.to_python_() for c in self.children))

    def to_z3_(self):
        return z3.AtMost(*[a.to_z3_() for a in self.children], 1)

    def vectorize_(self):
        return "(np.sum(({}), axis=0) <= 1)".format(", ".join(c.vectorize_() for c in self.children))


class ClassifierDescription(object):
    def __init__(self, input_space, output_spaces, instanciator=False):
        self.input_space = input_space
        self.output_spaces = output_spaces
        self.instanciator = instanciator

    def __repr__(self):
        return f"multiclass({self.input_space}, {self.output_spaces})"


class Multiclass(ClassifierDescription):
    pass


class Binary(ClassifierDescription):
    def __init__(self, input_space, output_space):
        super().__init__(input_space, [input_space & (~output_space), output_space])


class FullSpace(LabelSubspace):
    def vectorize_(self):
        return "True"

    def to_pyeda_(self):
        return pyeda.inter.expr(True)

    def to_z3_(self):
        return True

    def to_python_(self):
        return "True"

    def __repr__(self):
        return "full_space"


class Empty(LabelSubspace):
    def vectorize_(self):
        return "False"

    def to_pyeda_(self):
        return pyeda.inter.expr(False)

    def to_z3_(self):
        return False

    def to_python_(self):
        return "False"

    def __repr__(self):
        return "empty"


class ClassifierAtom(AtomLabelSubspace):
    @property
    def classifier_idx(self):
        return int(self.uid[1:].split('_')[0])

    @property
    def relative_idx(self):
        return int(self.uid[1:].split('_')[1])


class BratLabelAtom(AtomLabelSubspace):
    @property
    def annotation_type(self):
        if '__' in self.uid:
            return "attribute"
        else:
            return "mention"

    @property
    def annotation_name(self):
        return self.uid.split('__')[0]

    @property
    def value(self):
        if '__' in self.uid:
            return self.uid.split('__')[1]
        return None


class LabelFactory(object):
    def __init__(self):
        self.atoms = {}
        self.last_indexes = [0]

    def get_atom(self, name):
        if name not in self.atoms:
            if name.startswith("_"):
                atom = ClassifierAtom(name, self.last_indexes[-1], self)
            else:
                atom = BratLabelAtom(name, self.last_indexes[-1], self)
            self.atoms[name] = atom, self.last_indexes[-1]
            self.last_indexes[-1] += 1
            return atom
        else:
            return self.atoms[name][0]

    def from_pyeda(self, expr):
        return LabelSubspace.from_pyeda(expr, self)

    def __getitem__(self, name):
        return self.get_atom(name)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError()
        return self.get_atom(name)

    @contextlib.contextmanager
    def new_atoms_set(self):
        self.last_indexes.append(0)
        yield self
        self.last_indexes.pop(-1)


def rebase(expr, label_space, bases, factory=None):
    if factory is None:
        factory = expr.any_factory()
    cnf = (label_space & expr).to_cnf(simplify=False)
    bases = set((var for base in bases for var in base.support))

    # Propagate target through clauses to eliminate untouched ones
    clauses_to_explore = list(cnf.children)
    clauses_to_keep = []
    head = expr.support
    last_change = 0
    while len(clauses_to_explore):
        clause = clauses_to_explore.pop(0)
        if clause.support & head:
            head = head | clause.support
            clauses_to_keep.append(clause)
            last_change = 0
        else:
            clauses_to_explore.append(clause)
            last_change += 1
        if last_change > len(clauses_to_explore):
            break
    if len(clauses_to_keep) == 0:
        simplified = full_space
    else:
        # narrow_pruned = And(*(clauses[i] for i in first_clauses_indices))
        simplified = And(*clauses_to_keep)

        clauses = []
        for ass in simplified.satisfy_all(lib="pyeda"):
            clauses.append(And(*(symbol for symbol, val in ass.items() if val)))
        simplified = Or(*clauses)
        # Convert to dnf, and remove non _ literals, and then simplify with espresso
        clauses = []
        for clause in simplified.to_dnf(simplify=False).children:
            new_clause = []
            for clause_var in clause.children:
                if clause_var.support & bases:
                    new_clause.append(clause_var)
            if new_clause:
                clauses.append(And(*new_clause))
        simplified = factory.from_pyeda(espresso_exprs(Or(*clauses).to_pyeda())[0]).to_cnf()

        # Check whether clauses are equivalent or just sufficient
        sufficients = np.ones((len(simplified.children)), dtype=bool)
        equivalents = np.zeros((len(simplified.children)), dtype=bool)
        for i1, clause in enumerate(simplified.children):
            if not (label_space & ~(expr << clause)).satisfy_all():
                sufficients[i1] = 1
                # print(clause, "is sufficient")
                if not (label_space & ~(expr >> clause)).satisfy_all():
                    equivalents[i1] = 1
                    # print(clause, "is equivalent")
        if equivalents.any():
            simplified = And(*(clause for clause, flag in zip(simplified.children, equivalents) if flag))
        else:
            simplified = And(*(clause for clause, flag in zip(simplified.children, sufficients) if flag))
    return simplified


def make_base_converter(symbol_from_expr_mapping):
    """
    mapping: list of (symbol, expr)
    returns
        function(expr_mat: np.ndarray) -> symbol_mat: np.ndarray
    """
    vectorized = [None for _ in symbol_from_expr_mapping]
    for to_symbol, from_expr in symbol_from_expr_mapping:
        vectorized[to_symbol.absolute_idx] = from_expr.vectorize()

    def apply(expr_mat):
        return np.stack([fn(expr_mat) for fn in vectorized], 1)

    return apply


def preprocess_label_scheme(classifiers):
    """

    Parameters
    ----------
    classifiers: list of nlp.logic.ClassifierDescription

    Returns
    -------
    function, function, list, list
    """
    c = classifiers[0].output_spaces[0].any_factory()

    source_symbols = set()
    coded_classifiers_outputs = []
    with c.new_atoms_set():
        code_from_source_mapping = []  # [(code_symbol, source_expr)]
        for clf_i, clf in enumerate(classifiers):
            coded_clf_outputs = []
            for out_i, out in enumerate(clf.output_spaces):
                coded_out = c[f"_{clf_i}_{out_i}"]
                code_from_source_mapping.append((coded_out, out))
                source_symbols |= out.support
                source_symbols |= clf.input_space.support
                coded_clf_outputs.append(coded_out)
            coded_classifiers_outputs.append(coded_clf_outputs)
    # Sort symbols by their index
    source_symbols = sorted([symbol for symbol in source_symbols],
                            key=lambda x: next(s.index for s in source_symbols if s.name == x.name))
    coded_symbols = sorted([symbol for symbol, _ in code_from_source_mapping],
                           key=lambda x: next(s.index for s, _ in code_from_source_mapping if s.name == x.name))
    source_from_code_mapping = []  # [(source_symbol, code_expr)]

    # Express every classifier input (ex: measure_shape | measure_distance), as coded classes (_3_0 | _3_1)
    coded_classifiers_input = []
    for clf in classifiers:
        target_code_expr = rebase(
            clf.input_space,
            And(*(a == b for a, b in code_from_source_mapping)),
            coded_symbols)
        coded_classifiers_input.append(target_code_expr)

    # Express every source symbol (ex: measure_shape), as coded classes (ex: _3_0)
    # with consistency formula to respect hierarchy, exclusions etc (... & ~_2_1)

    for source_symbol in source_symbols:
        code_expr = rebase(
            source_symbol,
            And(*(a == b for a, b in code_from_source_mapping)),
            coded_symbols)

        head = code_expr.support
        new_code_expr = code_expr
        seen = set()
        while len(head):
            code_symbol = head.pop()
            for code_clf_input, code_clf_outputs in zip(coded_classifiers_input, coded_classifiers_outputs):
                if any(code_symbol is o for o in code_clf_outputs):
                    new_code_expr = c.from_sympy(
                        (new_code_expr & c.from_sympy(new_code_expr.to_sympy().subs(
                            {code_symbol.to_sympy(): code_clf_input.to_sympy()}))).to_sympy().simplify())
                    head |= code_clf_input.support - seen
            seen.add(code_symbol)
        source_from_code_mapping.append((source_symbol, c.from_sympy(new_code_expr.to_sympy().simplify())))

    code_from_source_translator = make_base_converter(code_from_source_mapping)  # function(source_expr_mat) -> codes_mat
    source_from_code_translator = make_base_converter(source_from_code_mapping)  # function(code_expr_mat) -> sources_mat
    code_from_code_translator = make_base_converter([(symbol, symbol) for symbol in coded_symbols])  # function(code_expr_mat) -> sources_mat
    # TODO code from noisy source et inv
    return code_from_source_translator, source_from_code_translator, code_from_code_translator, source_symbols, coded_symbols


def encode_labels(labels, code_from_source_translator, coded_symbols, source_symbols, atom_level, label_col_name="label"):
    """

    Parameters
    ----------
    labels: pd.DataFrame
        Columns:
            - label
    atom_level: str or list of str
        The id name of each unique atom being classified in the scheme (ex: mention_id)
    label_col_name: str
        The col name of the labels

    Returns
    -------

    """
    labels["_id"], unique_rows = factorize_rows(labels, subset=atom_level, return_categories=True)
    # Since we are going to overrwrite the label columnn, drop it from the original rows we're going to concatenate with
    # the output at the end of the function
    # unique_rows = unique_rows.drop(columns=[label_col_name])

    labels = labels[labels[label_col_name].isin([s.name for s in source_symbols])].copy()
    labels[label_col_name] = labels[label_col_name].astype(pd.CategoricalDtype([s.name for s in source_symbols]))
    new_labels = df_to_csr(labels["_id"], labels[label_col_name])
    new_labels = csr_matrix(code_from_source_translator(new_labels.toarray()))
    new_labels = csr_to_df(new_labels, row_name="_id", col_name=label_col_name)
    new_labels[label_col_name] = pd.Categorical.from_codes(new_labels[label_col_name], categories=[s.name for s in coded_symbols])
    new_labels['classifier_idx'] = new_labels[label_col_name].cat.codes.apply(lambda label: coded_symbols[label].classifier_idx)
    new_labels['relative_idx'] = new_labels[label_col_name].cat.codes.apply(lambda label: coded_symbols[label].relative_idx)
    new_labels = pd.merge(
        new_labels.reset_index(drop=True),
        unique_rows.assign(_id=np.arange(len(unique_rows)))
    ).drop(columns=["_id"])
    return new_labels


multiclass = Multiclass
binary = Binary
full_space = FullSpace()
empty = Empty()