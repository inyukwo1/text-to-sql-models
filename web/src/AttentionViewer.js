import React, { Component } from 'react';
import {BootstrapTable, 
       TableHeaderColumn} from 'react-bootstrap-table';
import './App.css';
import 'react-bootstrap-table/css/react-bootstrap-table.css'
 
 
class AttentionViewer extends Component {
  render() {
    const question_headers = []
    for (const [index, value] of this.props.question.entries()) {
      question_headers.push(<TableHeaderColumn row='1' dataField={index} width='100' >{value}</TableHeaderColumn>)
    }
    return (
      <div>
        <BootstrapTable data={this.props.data}>
          <TableHeaderColumn row='0' rowSpan='2' dataField='rule' width='300' isKey>
            Rules
          </TableHeaderColumn>
          <TableHeaderColumn row='0' colSpan={this.props.question.length}>
            Attention
          </TableHeaderColumn>
          {question_headers}
        </BootstrapTable>
      </div>
    );
  }
}
 
export default AttentionViewer;